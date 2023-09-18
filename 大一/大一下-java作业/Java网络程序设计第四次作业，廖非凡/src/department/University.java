package department;

import java.util.Vector;

import main.Stage;
import person.Ruler;

public class University extends Stage {
	Vector<Academy> acList = new Vector();
	Vector<Admin> adminList = new Vector();

	public University() {
	}

	public University(String name2, Ruler boss2, Academy firstAc, Admin firstAdim) {
		name = name2;
		boss2.level = Ruler.L1;
		boss = boss2;
		AddAcademy(firstAc);
		adminList.add(firstAdim);

		ShowMe();
	}

	public University(String name2, Ruler boss2, Vector<Academy> acList2, Admin firstAdim) {
		name = name2;
		boss2.level = Ruler.L1;
		boss = boss2;

		for (Academy s : acList2) {
			AddAcademy(s);
		}
		adminList.add(firstAdim);

		ShowMe();
	}

	public boolean AddAcademy(Academy academy) {
		for (Academy s : acList) {
			if (s == academy)
				return false;
		}
		acList.add(academy);
		return true;
	}

	@Override

	public void ShowMe() {
		System.out.println();
		System.out.println("\nUniversity:" + name + "\nPrisident:" + boss.name);
		System.out.println("Academy:");
		for (Academy s : acList) {
			System.out.println("“" + s.name + "”");
		}
		System.out.println("Admin:");
		for (Admin s : adminList) {
			System.out.println("“" + s.name + "”");
		}

	}

}
